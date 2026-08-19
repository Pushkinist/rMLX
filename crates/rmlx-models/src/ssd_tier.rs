//! SSD prompt-cache tier — arch dispatch shim.
//!
//! The SSD-tier machinery (config install, per-namespace startup maintenance,
//! pre-release v1 wipe, layout-key compute, spill / hydrate / block-I/O /
//! index modules + the 5 Prometheus hook globals) lives in `rmlx-kv-ssd`.
//! `rmlx-kv-ssd` cannot reach back into `rmlx-models`, so the arch-specific
//! dispatch — calling each arch's `PROMPT_CACHE` static's `attach_ssd_tier`
//! method directly — lives here alongside [`rmlx_kv_ssd::prepare_attach`].
//!
//! The `pub use rmlx_kv_ssd::ssd_tier::*` shim was dropped — callers in
//! `rmlx-cli` / `rmlx-server` import `SsdTierConfig`, `install_config`,
//! `active`, `compute_layout_key` directly from `rmlx_kv_ssd::ssd_tier::*`.

use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;

use crate::kv_cache::{kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};

/// At model-load, run startup maintenance (prune + evict-to-budget) for the
/// namespace and attach the spiller + hydrator onto the right per-arch
/// `PROMPT_CACHE`. No-op when the tier is OFF.
///
/// `(n_layers, n_kv_heads, head_dim)` are taken from the loaded model's config.
/// `n_layers` is expanded here into the **effective per-layer codec vector**
/// via [`kv_quant_for_layer`] — the same call every arch's cache construction
/// makes, with the same `LAYER_ADAPTIVE_*` constants — and that vector, not the
/// base codec alone, is what [`rmlx_kv_ssd::prepare_attach`] folds into the
/// `layout_key` the spiller/hydrator carry. The boundary-layer promotion is a
/// policy that can change between builds; a block written under one mixture
/// must not hydrate into a request running another, and the key is the only
/// thing standing between them.
///
/// `arch` must be the **resolved** class (`Architecture::arch_class()`), not
/// the checkpoint's declared `architectures[0]`: it both selects the per-arch
/// `PROMPT_CACHE` and salts the `layout_key`, so a declared name that does not
/// describe the model that was built would pick the wrong cache (or none). The
/// arms below still tolerate a declared alias for any caller that passes one.
///
/// No model identity is threaded here on purpose. The per-arch attach slot
/// holds one set of parameters and the last load wins, so a per-model value
/// recorded at attach would be wrong for every other resident model of the
/// arch. The hydrate probe takes the requesting model's seed from the request
/// instead.
///
/// The per-namespace SSD work (maintenance, layout-key compute,
/// logging) lives in [`rmlx_kv_ssd::prepare_attach`]; only the per-arch
/// `attach_ssd_tier` dispatch remains here because the trait impls
/// (`SsdSpiller: SpillSink<…>`, `SsdHydrator: SsdHydrate<…>`) live in
/// `rmlx-models`.
pub fn attach_at_load(
    arch: &str,
    model_id: &str,
    kv_quant: Option<KvQuant>,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: Device,
) {
    let layer_quants: Vec<KvQuant> = kv_quant
        .map(|base| {
            (0..n_layers)
                .map(|i| {
                    kv_quant_for_layer(
                        i,
                        n_layers,
                        base,
                        LAYER_ADAPTIVE_TAIL_N,
                        LAYER_ADAPTIVE_HEAD_N,
                    )
                })
                .collect()
        })
        .unwrap_or_default();

    let Some(info) = rmlx_kv_ssd::prepare_attach(
        arch,
        model_id,
        kv_quant,
        &layer_quants,
        n_kv_heads,
        head_dim,
        device,
    ) else {
        return; // tier OFF, kv_quant unresolved, or empty layout — logged inside prepare_attach
    };

    match arch {
        "Gemma4ForConditionalGeneration" | "Gemma4UnifiedForConditionalGeneration" => {
            crate::gemma4::prompt_cache::PROMPT_CACHE.attach_ssd_tier(
                &info.namespace,
                info.kv_quant,
                info.layout_key,
                info.device,
            );
        }
        "Gemma3ForConditionalGeneration" => {
            crate::gemma3::prompt_cache::PROMPT_CACHE.attach_ssd_tier(
                &info.namespace,
                info.kv_quant,
                info.layout_key,
                info.device,
            );
        }
        // Both Qwen3.5 shapes share one loader, one model struct and one
        // PROMPT_CACHE static, so the dense class gets the tier too. Listing
        // only the MoE name left every dense Qwen3.5 snapshot silently RAM-only.
        "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5ForConditionalGeneration" => {
            crate::qwen3_5_moe::prompt_cache::PROMPT_CACHE.attach_ssd_tier(
                &info.namespace,
                info.kv_quant,
                info.layout_key,
                info.device,
            );
        }
        "Qwen3ForCausalLM" => {
            crate::qwen3::QWEN3_PROMPT_CACHE.attach_ssd_tier(
                &info.namespace,
                info.kv_quant,
                info.layout_key,
                info.device,
            );
        }
        "Qwen2ForCausalLM" => {
            crate::qwen2::prompt_cache::PROMPT_CACHE.attach_ssd_tier(
                &info.namespace,
                info.kv_quant,
                info.layout_key,
                info.device,
            );
        }
        "BitNetForCausalLM" => {
            crate::bitnet::prompt_cache::PROMPT_CACHE.attach_ssd_tier(
                &info.namespace,
                info.kv_quant,
                info.layout_key,
                info.device,
            );
        }
        "Qwen3VLMoeForConditionalGeneration" => {
            crate::qwen3_vl_moe::prompt_cache::PROMPT_CACHE.attach_ssd_tier(
                &info.namespace,
                info.kv_quant,
                info.layout_key,
                info.device,
            );
        }
        "LagunaForCausalLM" => {
            crate::laguna::prompt_cache::PROMPT_CACHE.attach_ssd_tier(
                &info.namespace,
                info.kv_quant,
                info.layout_key,
                info.device,
            );
        }
        other => {
            tracing::info!(
                arch = other,
                namespace = %info.namespace,
                "SSD tier enabled but this arch has no spill/hydrate impl — prompt cache stays RAM-only"
            );
        }
    }
}
