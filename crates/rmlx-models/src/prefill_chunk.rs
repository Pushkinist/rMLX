//! Per-architecture prefill chunk sizing.
//!
//! Each architecture's `generate_greedy` processes the prompt in fixed-size
//! chunks. The chunk size trades off three things:
//!
//! - **Cold TTFT** — larger chunks amortize per-chunk FFI + Metal
//!   command-buffer flush overhead, lowering the time-to-first-token on
//!   long prompts.
//! - **Metal GPU watchdog** — chunks too large blow past the ~10 s
//!   command-buffer budget per forward pass. The cost is the per-chunk
//!   full-attention + KV work; Qwen3.5MoE's GatedDeltaNet runs the
//!   `gated_delta_step_gpu` kernel (T-loop in registers, grid independent of
//!   T) so it scales gracefully with chunk size rather than exploding a lazy
//!   graph.
//! - **First-call compile cost** — `compile_shapeless` traces a fresh
//!   Metal program per unique input shape; bigger chunks mean a heavier
//!   first-call trace before warmup populates the cache.
//!
//! The defaults below are tuned per-arch from follow-up bench data.
//! Override at runtime with `RMLX_PREFILL_CHUNK` (global) or
//! `RMLX_PREFILL_CHUNK_<ARCH>` (per-arch, arch upper-cased), e.g.
//! `RMLX_PREFILL_CHUNK_QWEN3_5_MOE=128`. Resolution order:
//!
//! runtime override (`set_prefill_chunk`) > per-arch env > global env > arch default > 64 fallback
//!
//! ## Runtime override (adaptive prefill chunk)
//!
//! [`set_prefill_chunk`] installs a process-wide runtime override stored in an
//! `AtomicUsize`. The override applies to ALL architectures and is read
//! lock-free on every prefill step. It is intended exclusively for the
//! adaptive admission controller. `0` means "no override" (default).

#![allow(clippy::match_same_arms)]
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};

const FALLBACK: usize = 64;

/// Minimum value accepted by [`set_prefill_chunk`].
pub const PREFILL_CHUNK_MIN: usize = 32;
/// Maximum value accepted by [`set_prefill_chunk`].
pub const PREFILL_CHUNK_MAX: usize = 2048;

/// Process-wide runtime prefill-chunk override.
///
/// `0` = unset (use normal resolution order).
/// Set by [`set_prefill_chunk`]; read lock-free by [`prefill_chunk_for`].
static RUNTIME_OVERRIDE: AtomicUsize = AtomicUsize::new(0);

/// Install a process-wide runtime override for the prefill chunk size.
///
/// The value is clamped to `[PREFILL_CHUNK_MIN, PREFILL_CHUNK_MAX]`.
/// Pass `0` to clear the override and revert to the normal resolution order.
///
/// # Lock-free contract
///
/// This stores with `Release` ordering. Readers in [`prefill_chunk_for`] use
/// `Acquire`, which guarantees that any subsequent `prefill_chunk_for` call on
/// any thread will observe the new value.
pub fn set_prefill_chunk(value: usize) {
    let clamped = if value == 0 {
        0
    } else {
        value.clamp(PREFILL_CHUNK_MIN, PREFILL_CHUNK_MAX)
    };
    RUNTIME_OVERRIDE.store(clamped, Ordering::Release);
}

/// Read the current runtime override without the normal resolution fallback.
///
/// Returns `None` when no override is active (`RUNTIME_OVERRIDE == 0`).
pub fn runtime_override() -> Option<usize> {
    let v = RUNTIME_OVERRIDE.load(Ordering::Acquire);
    if v == 0 {
        None
    } else {
        Some(v)
    }
}

/// Returns the prefill chunk size for `arch`. `arch` is the module-style
/// identifier (`"qwen3"`, `"qwen3_5_moe"`, `"gemma3"`, `"gemma4"`,
/// `"laguna"`, `"qwen2"`). New archs slot in by adding a row to
/// `arch_default()`.
pub fn prefill_chunk_for(arch: &str) -> usize {
    // Runtime override (adaptive prefill chunk) — highest precedence.
    let rt = RUNTIME_OVERRIDE.load(Ordering::Acquire);
    if rt != 0 {
        return rt;
    }

    let per_arch_var = format!("RMLX_PREFILL_CHUNK_{}", arch.to_uppercase());
    if let Some(v) = env::var(&per_arch_var).ok().and_then(|s| s.parse().ok()) {
        return v;
    }
    if let Some(v) = env::var("RMLX_PREFILL_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        return v;
    }
    arch_default(arch).unwrap_or(FALLBACK)
}

/// Map a config `architectures[0]` class name (e.g. `"Qwen3ForCausalLM"`) to
/// the module-style key understood by [`prefill_chunk_for`]. Unknown classes
/// return `""`, which `prefill_chunk_for` resolves to the conservative
/// `FALLBACK` chunk — never the oversized gemma4 default.
pub fn module_key_for_class(arch_class: &str) -> &'static str {
    match arch_class {
        "Gemma4ForConditionalGeneration" | "Gemma4UnifiedForConditionalGeneration" => "gemma4",
        "Gemma3ForConditionalGeneration" => "gemma3",
        "Qwen2ForCausalLM" => "qwen2",
        "Qwen3ForCausalLM" => "qwen3",
        "LagunaForCausalLM" => "laguna",
        "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5ForConditionalGeneration" => "qwen3_5_moe",
        "BitNetForCausalLM" => "bitnet",
        "Qwen3VLMoeForConditionalGeneration" => "qwen3_vl_moe",
        // Any unsupported class: no dedicated prefill-chunk row, fall through to
        // FALLBACK (safe, conservative).
        _ => "",
    }
}

fn arch_default(arch: &str) -> Option<usize> {
    match arch {
        "qwen3" => Some(256),
        // qwen3_5_moe: 2048, matching mlx-lm's prefill_step_size. The GDN
        // recurrence now always runs the `gated_delta_step_gpu` Metal kernel
        // (one dispatch, T-loop in registers) instead of flipping to the
        // ops-graph path at T>=256 — so a large chunk no longer explodes the
        // lazy graph; it just means fewer, bigger forward passes and far fewer
        // per-chunk KV-state evals. A real-model sweep on Qwen3.6-35B-A3B-8bit
        // (kv-none) measured TTFT improving monotonically 64→2048 with no Metal
        // watchdog trip: 4k 4240→1065ms (4.0x), 8k 9008→2136ms (4.2x), 16k
        // 19489→4712ms (4.1x); decode TPS unchanged (the decode kernel is the
        // same at every chunk size). Override via
        // `RMLX_PREFILL_CHUNK_QWEN3_5_MOE`.
        "qwen3_5_moe" => Some(2048),
        // qwen3_vl_moe: plain GQA MoE (no GDN linear attention), so it tolerates
        // the same large chunk as gemma4. Native image tiling produces thousands
        // of soft tokens; a single-shot forward over the full prompt trips the
        // Metal ~10s GPU watchdog, so the image prefill is chunked at 512.
        "qwen3_vl_moe" => Some(512),
        "gemma3" => Some(256),
        // gemma4 default 512: p0b-ttft bench measured -30% cold TTFT at 8K
        // and -12% at 32K vs chunk=64 with no Metal watchdog at max-ctx 64K.
        "gemma4" => Some(512),
        "qwen2" => Some(256),
        // laguna currently lacks tuning data; preserve its pre-tuning value
        // of 256 (was hardcoded as `PREFILL_CHUNK = 256` before this module
        // landed). Spec called for 64 but that would regress the
        // already-shipping value with no evidence; revisit when laguna
        // bench data exists.
        "laguna" => Some(256),
        // bitnet: max_position_embeddings=4096, no GDN prefill ops.
        // 64 is conservative but safe; no bench data yet to justify larger.
        "bitnet" => Some(64),
        _ => None,
    }
}

#[cfg(test)]
#[path = "prefill_chunk_tests.rs"]
mod tests;
