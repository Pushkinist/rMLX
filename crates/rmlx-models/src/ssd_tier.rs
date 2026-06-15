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

/// At model-load, run startup maintenance (prune + evict-to-budget) for the
/// namespace and attach the spiller + hydrator onto the right per-arch
/// `PROMPT_CACHE`. No-op when the tier is OFF.
///
/// `(n_layers, n_kv_heads, head_dim)` are taken from the loaded model's
/// config and folded into the `layout_key` that the spiller/hydrator carry.
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
    let Some(info) = rmlx_kv_ssd::prepare_attach(
        arch, model_id, kv_quant, n_layers, n_kv_heads, head_dim, device,
    ) else {
        return; // tier OFF or kv_quant unresolved — already logged inside prepare_attach
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
        "Qwen3_5MoeForConditionalGeneration" => {
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
        other => {
            tracing::info!(
                arch = other,
                namespace = %info.namespace,
                "SSD tier enabled but this arch has no spill/hydrate impl — prompt cache stays RAM-only"
            );
        }
    }
}
