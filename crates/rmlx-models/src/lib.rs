#![warn(missing_docs)]
//! Per-architecture inference graphs.
//!
//! Stage 1 wired architectures:
//! `gemma4` — Gemma4ForConditionalGeneration (mxfp8 quantized; 26B has MoE block).
//! `gemma3` — Gemma3ForConditionalGeneration (affine-int8 quantized, e.g. medgemma).
//! `qwen2` — Qwen2ForCausalLM (dense, affine quantized).
//! `qwen3` — Qwen3ForCausalLM (dense, affine quantized, adds per-head q/k RMSNorm).
//! `laguna` — LagunaForCausalLM (sparse MoE, mxfp8, per-tensor overrides).
//! `qwen3_5_moe` — Qwen3_5MoeForConditionalGeneration (hybrid GatedDeltaNet + full-attn, sparse MoE).
//!
//! `layers` — shared building blocks (RmsNorm, Linear, Embedding, Mlp, MoeBlock stub).
//! `arch` — architecture dispatch: load_model() reads config.architectures[0].

#![cfg_attr(
    test,
    allow(
        clippy::panic,
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
 // ignore_without_reason: integration test #[ignore] attributes in test modules
 // omit reasons; suppressed here rather than adding boilerplate to every test.
        clippy::ignore_without_reason,
    )
)]

pub mod arch;
pub mod audio_io;
pub mod bitnet;
pub mod block_manager;
pub mod calibration_sink;
pub mod constraint;
pub(crate) mod decode_loop;
pub mod gated_delta_msl;
pub mod gemma3;
pub mod gemma4;
pub mod jina_v4;
pub mod kv_cache;
pub mod laguna;
pub mod layers;
pub(crate) mod load_util;
pub mod multimodal_cache;
pub mod paroquant_msl;
pub mod ppl;
pub mod prefill_chunk;
pub mod prefix_index;
pub mod prompt_cache;
pub mod qwen2;
pub mod qwen3;
pub mod qwen3_5_moe;
pub mod qwen3_vl_moe;
pub mod rope;
pub mod sampler;
pub mod speculative;
pub mod ssd_tier;

// The MSL wrapper re-exports (`rmlx_models::{q8_msl,
// turboquant_msl, planarquant_msl, turbo_flash_msl, sparse_v_msl}`) were
// dropped — there were zero callers in the workspace. Import directly from
// `rmlx_kv_quant::{q8_msl,turboquant_msl,planarquant_msl,turbo_flash_msl,
// sparse_v_msl}` if needed.

pub use arch::{is_arch_supported, read_load_phases, LoadPhases};
pub use constraint::{ConstraintEngine, NoOpConstraint};
pub use decode_loop::{ProbeStep, SmokeVerdict};
// The flat `rmlx_models::{KvCache, KvQuant, LinearAttnCache,
// write_caches, set_ssd_*}` re-exports were dropped. Codec types live at
// `rmlx_kv_quant::*`, SSD-tier hooks + `write_caches` at `rmlx_kv_ssd::*`.
// Policy items (`KvCacheBuilder`, `ResolverSignals`, `kv_quant_for_ctx`,
// `kv_quant_for_layer`, `LAYER_ADAPTIVE_*`) stay in `rmlx_models::kv_cache`
// — import via `rmlx_models::kv_cache::*`.
// The SSD KV tier lives in `rmlx-kv-ssd`. The top-level
// `rmlx_models::ssd_tier` module is a thin arch dispatch shim that owns the
// per-arch `attach_at_load` switch and calls `rmlx_kv_ssd::prepare_attach`.
// The `pub use rmlx_kv_ssd as kv_ssd;` convenience alias was dropped —
// there were zero callers. Import directly from `rmlx_kv_ssd`.
pub use prompt_cache::{classify_kv_bytes, CacheStats, KvBytesSample, KvBytesVerdict};
pub use sampler::{Pcg32, PenaltyConfig, SamplerConfig, TokenLogprobs};
pub use speculative::{DraftKind, SpeculativeDispatcher};
